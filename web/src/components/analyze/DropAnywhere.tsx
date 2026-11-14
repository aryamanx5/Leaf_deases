import { ImageDown } from 'lucide-react';
import { useEffect, useState } from 'react';

/**
 * Lets a photo be dropped anywhere on the page, not only on the dashed box: while a file is dragged over the
 * window a full-page target appears. The drop zone itself keeps handling drops that land on it.
 */
export function DropAnywhere({ onFile }: { onFile: (f: File) => void }) {
  const [over, setOver] = useState(false);
  useEffect(() => {
    let depth = 0;
    const hasFile = (e: DragEvent) => [...(e.dataTransfer?.types ?? [])].includes('Files');
    const enter = (e: DragEvent) => {
      if (!hasFile(e)) return;
      depth++;
      setOver(true);
    };
    const leave = (e: DragEvent) => {
      if (!hasFile(e)) return;
      depth = Math.max(0, depth - 1);
      if (!depth) setOver(false);
    };
    const dragover = (e: DragEvent) => {
      if (hasFile(e)) e.preventDefault(); // allow the drop
    };
    const drop = (e: DragEvent) => {
      depth = 0;
      setOver(false);
      if (!hasFile(e) || e.defaultPrevented) return; // the dashed box already took it
      e.preventDefault();
      const f = [...(e.dataTransfer?.files ?? [])].find((x) => x.type.startsWith('image/'));
      if (f) onFile(f);
    };
    window.addEventListener('dragenter', enter);
    window.addEventListener('dragleave', leave);
    window.addEventListener('dragover', dragover);
    window.addEventListener('drop', drop);
    return () => {
      window.removeEventListener('dragenter', enter);
      window.removeEventListener('dragleave', leave);
      window.removeEventListener('dragover', dragover);
      window.removeEventListener('drop', drop);
    };
  }, [onFile]);
  if (!over) return null;
  return (
    <div aria-hidden="true" data-testid="drop-anywhere" className="veil-in pointer-events-none fixed inset-0 z-[250] flex items-center justify-center bg-ink/80 p-6 backdrop-blur-sm">
      <div className="flex h-full w-full flex-col items-center justify-center gap-4 rounded-[18px] border-2 border-dashed border-leaf">
        <ImageDown size={40} strokeWidth={1.2} className="float-y text-leaf" />
        <p className="text-[28px] tracking-[-0.03em] text-text sm:text-[40px]">
          Drop the leaf <span className="font-playfair italic">anywhere.</span>
        </p>
        <p className="mono text-[12px] text-muted">one photo · it is read straight away</p>
      </div>
    </div>
  );
}
