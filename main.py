import os
import json
import hashlib
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configurar pydub para encontrar ffmpeg
_ffmpeg_path = shutil.which("ffmpeg")
if not _ffmpeg_path:
    # Fallback: buscar en ubicación conocida de winget
    _winget_link = os.path.expanduser("~\\AppData\\Local\\Microsoft\\WinGet\\Links\\ffmpeg.exe")
    if os.path.exists(_winget_link):
        _ffmpeg_path = _winget_link
if _ffmpeg_path:
    os.environ["PATH"] = os.path.dirname(_ffmpeg_path) + os.pathsep + os.environ.get("PATH", "")
    from pydub import AudioSegment
    AudioSegment.converter = _ffmpeg_path
    _ffprobe_path = shutil.which("ffprobe")
    if _ffprobe_path:
        AudioSegment.ffprobe = _ffprobe_path
    print(f"[STARTUP] ffmpeg: {_ffmpeg_path}")
else:
    print("[STARTUP] WARNING: ffmpeg NOT FOUND!")

app = FastAPI(title="DJ Waveform API", version="1.0.0")

# CORS para desarrollo local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
SAMPLES_COUNT = 800  # Muestras del waveform (resolución horizontal)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("waveform")


def get_cache_path(video_id: str) -> Path:
    """Ruta del archivo cache para un video ID."""
    return CACHE_DIR / f"{video_id}.json"


def load_from_cache(video_id: str) -> Optional[dict]:
    """Carga waveform desde cache si existe."""
    path = get_cache_path(video_id)
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def save_to_cache(video_id: str, data: dict):
    """Guarda waveform en cache."""
    path = get_cache_path(video_id)
    with open(path, "w") as f:
        json.dump(data, f)


def extract_video_id(url_or_id: str) -> str:
    """Extrae video ID de una URL de YouTube o devuelve el ID directamente."""
    import re
    # Patrones comunes de URL de YouTube
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|music\.youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',  # ID directo
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return url_or_id


def download_and_analyze(video_id: str) -> dict:
    """Descarga audio de YouTube y genera datos de waveform."""
    import yt_dlp
    import shutil

    # Asegurar que ffmpeg es encontrable
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        logger.info(f"ffmpeg encontrado: {ffmpeg_path}")

    url = f"https://www.youtube.com/watch?v={video_id}"
    logger.info(f"Descargando audio para {video_id}...")

    # Descargar solo audio
    tmp_dir = tempfile.mkdtemp()
    try:
        output_path = os.path.join(tmp_dir, "audio.%(ext)s")
        ydl_opts = {
            'format': 'worstaudio/worst',  # worstaudio=más rápido (solo necesitamos FFT), worst=fallback con audio+video
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            },
        }

        # Indicar ubicación de ffmpeg si está disponible
        if ffmpeg_path:
            ydl_opts['ffmpeg_location'] = os.path.dirname(ffmpeg_path)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
        except Exception as e:
            logger.error(f"Error descargando {video_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Error descargando audio: {str(e)}")

        # Buscar el archivo de audio generado
        audio_file = None
        for ext in ['wav', 'webm', 'opus', 'm4a', 'mp3', 'ogg', 'mp4']:
            candidate = os.path.join(tmp_dir, f"audio.{ext}")
            if os.path.exists(candidate):
                audio_file = candidate
                break
        if audio_file is None:
            files = [f for f in Path(tmp_dir).glob("audio*") if f.is_file()]
            if files:
                audio_file = str(files[0])
            else:
                raise HTTPException(status_code=500, detail="No se generó archivo de audio")

        logger.info(f"Archivo de audio: {audio_file} ({os.path.getsize(audio_file)} bytes)")
        logger.info(f"Analizando audio de '{title}' ({duration}s)...")
        waveform = analyze_audio(audio_file)

    finally:
        # Limpiar directorio temporal manualmente
        import shutil as _shutil
        try:
            _shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    result = {
        "id": video_id,
        "title": title,
        "duration": duration,
        "samples": SAMPLES_COUNT,
        "waveform": waveform,
    }

    return result


def analyze_audio(audio_path: str) -> list:
    """Analiza un archivo de audio y devuelve datos de waveform normalizados.
    
    Genera SAMPLES_COUNT muestras, cada una representando la amplitud
    RMS de un segmento del audio, con énfasis en frecuencias bajas.
    Soporta WAV, WebM, Opus, M4A, MP3 via pydub+ffmpeg.
    """
    from pydub import AudioSegment

    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_sample_width(2)  # Mono, 16-bit
        raw_data = audio.raw_data
        framerate = audio.frame_rate
    except Exception as e:
        import traceback
        logger.error(f"Error leyendo audio: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error analizando audio: {str(e)}")

    # Convertir bytes a numpy float32
    samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)

    if len(samples) == 0:
        return [0.0] * SAMPLES_COUNT

    # Dividir en SAMPLES_COUNT segmentos y calcular RMS con énfasis en bajos
    chunk_size = max(1, len(samples) // SAMPLES_COUNT)
    waveform = []

    for i in range(SAMPLES_COUNT):
        start = i * chunk_size
        end = min(start + chunk_size, len(samples))
        if start >= len(samples):
            waveform.append(0.0)
            continue

        chunk = samples[start:end]

        # FFT para extraer bajos (0-200Hz)
        if len(chunk) >= 64:
            fft = np.abs(np.fft.rfft(chunk))
            freq_bins = np.fft.rfftfreq(len(chunk), d=1.0 / framerate)

            # Máscara de frecuencias bajas (0-200Hz = bass)
            bass_mask = freq_bins <= 200
            mid_mask = (freq_bins > 200) & (freq_bins <= 2000)

            bass_energy = np.mean(fft[bass_mask]) if np.any(bass_mask) else 0
            mid_energy = np.mean(fft[mid_mask]) if np.any(mid_mask) else 0

            # Combinar: 70% bajos + 30% medios
            combined = bass_energy * 0.7 + mid_energy * 0.3
        else:
            combined = np.sqrt(np.mean(chunk ** 2))  # RMS simple

        waveform.append(float(combined))

    # Normalizar a 0.0-1.0
    max_val = max(waveform) if waveform else 1
    if max_val > 0:
        waveform = [round(v / max_val, 4) for v in waveform]

    return waveform


@app.get("/")
async def root():
    """Health check."""
    cache_count = len(list(CACHE_DIR.glob("*.json")))
    return {"status": "ok", "service": "DJ Waveform API", "cached_tracks": cache_count}


@app.get("/waveform")
async def get_waveform(v: str = Query(..., description="YouTube video ID o URL")):
    """Obtiene datos de waveform para un video de YouTube.
    
    Devuelve un array de 800 muestras normalizadas (0.0-1.0)
    representando la amplitud con énfasis en frecuencias bajas.
    """
    video_id = extract_video_id(v)

    if not video_id or len(video_id) < 5:
        raise HTTPException(status_code=400, detail="Video ID inválido")

    # Verificar cache
    cached = load_from_cache(video_id)
    if cached:
        logger.info(f"Cache hit para {video_id}")
        return JSONResponse(content=cached)

    # Descargar y analizar
    logger.info(f"Cache miss para {video_id}, descargando...")
    result = download_and_analyze(video_id)

    # Guardar en cache
    save_to_cache(video_id, result)
    logger.info(f"Waveform generado y cacheado para {video_id}")

    return JSONResponse(content=result)


@app.delete("/cache/{video_id}")
async def clear_cache(video_id: str):
    """Elimina waveform cacheado de un video."""
    path = get_cache_path(video_id)
    if path.exists():
        path.unlink()
        return {"status": "deleted", "id": video_id}
    raise HTTPException(status_code=404, detail="No encontrado en cache")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
