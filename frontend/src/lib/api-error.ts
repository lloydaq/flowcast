/** An API failure: the API's own reason (its "detail") and the HTTP status (0 when unreachable). */
export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}
