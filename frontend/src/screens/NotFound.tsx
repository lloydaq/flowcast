import { Link } from "react-router";
import { useReplay } from "../lib/replay";

export function NotFound() {
  const r = useReplay();
  return (
    <div className="title-row">
      <div>
        <h1>Page not found</h1>
        <p className="finding">
          There is nothing at this address. <Link to={r.href("/")}>Back to the network</Link>.
        </p>
      </div>
    </div>
  );
}
