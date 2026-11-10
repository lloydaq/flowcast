import { useLayoutEffect, useRef, useState } from "react";

interface Option<T> {
  value: T;
  label: string;
}

/** A pill track whose white selected pill slides to the chosen segment (240 ms, ease-out). */
export function Segmented<T extends string | number>({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: readonly Option<T>[];
  value: T;
  onChange: (v: T) => void;
}) {
  const track = useRef<HTMLDivElement>(null);
  const [thumb, setThumb] = useState<{ x: number; w: number } | null>(null);

  useLayoutEffect(() => {
    const el = track.current;
    if (!el) return;
    const place = () => {
      const i = options.findIndex((o) => o.value === value);
      const btn = el.querySelectorAll("button")[i];
      if (btn) setThumb({ x: btn.offsetLeft, w: btn.offsetWidth });
    };
    place();
    const ro = new ResizeObserver(place);
    ro.observe(el);
    return () => ro.disconnect();
  }, [options, value]);

  return (
    <div className="segmented" role="group" aria-label={label} ref={track}>
      {thumb && (
        <span className="segmented-thumb" style={{ width: thumb.w, transform: `translateX(${thumb.x}px)` }} />
      )}
      {options.map((o) => (
        <button key={String(o.value)} type="button" aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
