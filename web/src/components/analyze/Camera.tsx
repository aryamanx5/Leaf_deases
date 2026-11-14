import { Camera as CameraIcon, RefreshCcw, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { Picked } from '@/components/analyze/Input';

type Status = 'starting' | 'live' | 'denied' | 'error';
const MAX_SIDE = 1600; // the captured frame is scaled so its longer side is at most this

const cameraSupported = () => typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia;

/**
 * "Use camera": opens the device camera in a dialog with a viewfinder, takes one frame and hands it to the same
 * path as an uploaded file. The stream is stopped the moment the dialog closes; nothing is recorded.
 */
export function CameraCapture({ onPick, disabled }: { onPick: (p: Picked) => void; disabled?: boolean }) {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<Status>('starting');
  const [facing, setFacing] = useState<'environment' | 'user'>('environment');
  const [many, setMany] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const opener = useRef<HTMLButtonElement>(null);
  const shutter = useRef<HTMLButtonElement>(null);

  const stop = useCallback(() => {
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
  }, []);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    setStatus('starting');
    navigator.mediaDevices
      .getUserMedia({ audio: false, video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } } })
      .then(async (s) => {
        if (!alive) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream.current = s;
        const v = videoRef.current;
        if (v) {
          v.srcObject = s;
          await v.play().catch(() => {});
        }
        setStatus('live');
        shutter.current?.focus();
        const cams = (await navigator.mediaDevices.enumerateDevices().catch(() => [])).filter((d) => d.kind === 'videoinput');
        if (alive) setMany(cams.length > 1);
      })
      .catch((e: DOMException) => {
        if (alive) setStatus(e.name === 'NotAllowedError' || e.name === 'SecurityError' ? 'denied' : 'error');
      });
    return () => {
      alive = false;
      stop();
    };
  }, [open, facing, stop]);

  const close = useCallback(() => {
    setOpen(false);
    stop();
    opener.current?.focus();
  }, [stop]);

  const capture = () => {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return;
    const k = Math.min(1, MAX_SIDE / Math.max(v.videoWidth, v.videoHeight));
    const c = document.createElement('canvas');
    c.width = Math.round(v.videoWidth * k);
    c.height = Math.round(v.videoHeight * k);
    c.getContext('2d')!.drawImage(v, 0, 0, c.width, c.height);
    c.toBlob((blob) => {
      if (!blob) return;
      onPick({ blob, url: URL.createObjectURL(blob), name: 'camera.jpg' });
      close();
    }, 'image/jpeg', 0.92);
  };

  if (!cameraSupported()) return null;
  return (
    <>
      <button ref={opener} type="button" disabled={disabled} onClick={() => setOpen(true)} data-testid="camera-open"
              className="inline-flex min-h-[44px] items-center gap-2 rounded-full border border-white/25 px-4 text-[13px] text-text transition-colors hover:border-white/60 hover:bg-white/5 disabled:opacity-50">
        <CameraIcon size={16} strokeWidth={1.5} aria-hidden="true" /> Use camera
      </button>
      {open && (
        <div className="fixed inset-0 z-[300] flex items-center justify-center p-4" data-testid="camera-dialog"
             onKeyDown={(e) => {
               if (e.key === 'Escape') close();
             }}>
          <div className="veil-in absolute inset-0 bg-ink/85 backdrop-blur-sm" onClick={close} aria-hidden="true" />
          <div role="dialog" aria-modal="true" aria-labelledby="camera-title"
               className="palette-in relative flex max-h-full w-full max-w-[720px] flex-col overflow-hidden rounded-[14px] border border-white/15 bg-soil">
            <div className="flex items-center justify-between gap-4 border-b hairline px-5 py-3">
              <h2 id="camera-title" className="text-[16px] text-text">Photograph a leaf</h2>
              <button type="button" onClick={close} aria-label="Close camera"
                      className="flex h-11 w-11 items-center justify-center rounded-full text-muted transition-colors hover:bg-white/10 hover:text-text">
                <X size={18} aria-hidden="true" />
              </button>
            </div>
            <div className="relative aspect-[4/3] w-full bg-ink">
              <video ref={videoRef} muted playsInline aria-label="Live camera preview" data-testid="camera-video"
                     className={`h-full w-full object-cover ${facing === 'user' ? '-scale-x-100' : ''}`} />
              {status === 'live' && (
                <div aria-hidden="true" className="pointer-events-none absolute inset-[9%]">
                  {['left-0 top-0 border-l-2 border-t-2', 'right-0 top-0 border-r-2 border-t-2', 'bottom-0 left-0 border-b-2 border-l-2', 'bottom-0 right-0 border-b-2 border-r-2'].map((c) => (
                    <span key={c} className={`absolute h-9 w-9 rounded-[3px] border-white/85 ${c}`} />
                  ))}
                </div>
              )}
              {status !== 'live' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-8 text-center" role="status">
                  {status === 'starting' && <p className="mono text-[13px] text-muted">Starting the camera…</p>}
                  {status === 'denied' && (
                    <>
                      <p className="text-[18px] text-text">The camera is blocked for this page</p>
                      <p className="max-w-[44ch] text-[14px] leading-relaxed text-muted">
                        Allow camera access in your browser’s address bar and try again, or close this and upload a photo instead.
                      </p>
                    </>
                  )}
                  {status === 'error' && (
                    <>
                      <p className="text-[18px] text-text">No camera could be started</p>
                      <p className="max-w-[44ch] text-[14px] leading-relaxed text-muted">
                        It may be in use by another app, or this device has none. You can upload a photo instead.
                      </p>
                    </>
                  )}
                </div>
              )}
            </div>
            <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-4">
              <p className="min-w-0 flex-1 basis-[220px] text-[13px] leading-snug text-muted">
                One leaf, close up, filling the corners. Daylight, no flash. The frame is sent only when you press the button.
              </p>
              <div className="flex items-center gap-2">
                {many && (
                  <button type="button" onClick={() => setFacing((f) => (f === 'environment' ? 'user' : 'environment'))} aria-label="Switch camera"
                          className="flex h-11 w-11 items-center justify-center rounded-full border border-white/25 text-text transition-colors hover:bg-white/10">
                    <RefreshCcw size={16} aria-hidden="true" />
                  </button>
                )}
                <button ref={shutter} type="button" onClick={capture} disabled={status !== 'live'} data-testid="camera-shutter"
                        className="inline-flex min-h-[44px] items-center gap-2 rounded-full bg-cta px-6 text-sm font-medium text-white transition-colors hover:bg-cta-hover disabled:opacity-50">
                  <CameraIcon size={16} aria-hidden="true" /> Take photo
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
