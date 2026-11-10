import { useEffect, useRef, useState } from "react";
import { int } from "../lib/format";
import { prefersReducedMotion } from "../lib/theme";

/** A whole number that counts up once when it first appears; later changes show at once. */
export function CountUp({ value }: { value: number }) {
  const first = useRef(true);
  const [shown, setShown] = useState(() => (prefersReducedMotion() ? value : 0));
  useEffect(() => {
    if (!first.current || prefersReducedMotion()) {
      setShown(value);
      return;
    }
    first.current = false;
    const start = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const p = Math.min((now - start) / 600, 1);
      setShown(Math.round(value * (1 - (1 - p) ** 3))); // ease-out; always ends on the true value
      if (p < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value]);
  return <>{int(shown)}</>;
}
