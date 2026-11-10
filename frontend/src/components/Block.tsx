import type { UseQueryResult } from "@tanstack/react-query";
import { CircleAlert } from "lucide-react";
import type { ReactNode } from "react";

/**
 * One block's data states (architect.md §10.8): while loading, the final layout with flat grey
 * placeholders; on failure, what failed and a Retry inside the block; otherwise the content.
 */
export function Block<T>({
  query,
  what,
  placeholder,
  children,
}: {
  query: UseQueryResult<T>;
  what: string; // "the network forecast"
  placeholder: ReactNode;
  children: (data: T) => ReactNode;
}) {
  if (query.data !== undefined) return <>{children(query.data)}</>;
  if (query.isError) {
    return (
      <div className="block-error" role="alert">
        <CircleAlert aria-hidden="true" />
        <span>
          Couldn’t load {what}: {query.error.message}.
        </span>
        <button type="button" className="text-button" onClick={() => void query.refetch()}>
          Retry
        </button>
      </div>
    );
  }
  return (
    <div aria-busy="true" aria-label={`Loading ${what}`}>
      {placeholder}
    </div>
  );
}

/** A flat grey bar standing in for text or a figure at its real size. No shimmer. */
export function Ph({ w, h = 14, style }: { w: number | string; h?: number; style?: React.CSSProperties }) {
  return <span className="ph" style={{ width: w, height: h, ...style }} />;
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="block-empty">{children}</p>;
}
