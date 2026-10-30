"""Field-level validation errors for autopilot targets, in words the target form can show next to the field.

* ``FieldError(field, msg)``: a ``ValueError`` a model validator raises to blame one field (``field`` is relative
  to the model that raises it; ``plain_errors`` prefixes the model's location, so ``HeadsetTarget`` raising
  ``FieldError("output_root", ...)`` becomes ``headset.output_root``).
* ``plain_errors(ValidationError)`` -> ``[{field, message}]``: pydantic's prefixes stripped, its bound messages
  rewritten (``ge`` -> "must be at least 1"), the location dotted.
* ``error_body(errors, fields=True)``: the 422 body every autopilot target route returns, ``{errors, detail}``
  (``detail`` joins them, for clients that show one line; ``fields=False`` leaves the field paths out of it, for a
  route whose caller shows the line as is, e.g. the head-set heads list).
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError


class FieldError(ValueError):
    def __init__(self, field: str, msg: str):
        super().__init__(msg)
        self.field, self.msg = field, msg


def _loc(loc: tuple) -> str:
    return ".".join(str(x) for x in loc if not (isinstance(x, str) and x.startswith("function-")))


def plain_message(e: dict[str, Any]) -> str:
    """One pydantic error dict in the form's words."""
    t, ctx = e.get("type", ""), e.get("ctx") or {}
    if t in ("greater_than_equal", "greater_than"):
        return f"must be at least {ctx.get('ge', ctx.get('gt'))}" if t == "greater_than_equal" else \
            f"must be more than {ctx.get('gt')}"
    if t in ("less_than_equal", "less_than"):
        return f"must be at most {ctx.get('le', ctx.get('lt'))}" if t == "less_than_equal" else \
            f"must be less than {ctx.get('lt')}"
    if t == "missing":
        return "is required"
    if t == "too_short":
        return "needs at least one entry" if ctx.get("min_length") == 1 else f"needs at least {ctx.get('min_length')}"
    if t == "literal_error":
        return f"must be one of {ctx.get('expected')}"
    if t in ("int_parsing", "int_type", "int_from_float"):
        return "must be a whole number"
    if t in ("float_parsing", "float_type"):
        return "must be a number"
    if t == "extra_forbidden":
        return "is not a setting this source has"
    msg = str(e.get("msg", ""))
    for prefix in ("Value error, ", "Assertion failed, "):
        msg = msg.removeprefix(prefix)
    return msg


def plain_errors(err: ValidationError) -> list[dict[str, str]]:
    out = []
    for e in err.errors(include_url=False):
        exc = (e.get("ctx") or {}).get("error")
        loc = _loc(tuple(e.get("loc") or ()))
        if isinstance(exc, FieldError):
            field = ".".join(x for x in (loc, exc.field) if x)
            out.append({"field": field, "message": exc.msg})
        else:
            out.append({"field": loc, "message": plain_message(e)})
    return out


def error_body(errors: list[dict[str, str]], fields: bool = True) -> dict:
    return {"errors": errors, "detail": "; ".join(f"{e['field']}: {e['message']}" if e["field"] and fields
                                                   else e["message"] for e in errors)}
