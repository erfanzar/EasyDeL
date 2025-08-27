from eformer.loggings import get_logger

logger = get_logger("Decoders")


class SmartBytecodeDecoder:
    """A smart decoder that handles partial token sequences and recovers from malformed characters.

    Optimized for performance with:
    - Unified recovery logic to reduce code duplication
    - Binary search for finding valid decode points
    - Caching of recent decode results
    - Fast malformed character detection
    """

    def __init__(self, processor, fallback_char: str = "", cache_size: int = 16):
        self.processor = processor
        self.fallback_char = fallback_char
        # Use frozenset for O(1) lookups instead of iterating
        self.malformed_indicators = frozenset({"�", "\\ufffd", "\ufffd"})  # noqa
        # Simple LRU cache for recent successful decode points
        self._decode_cache = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def contains_malformed_chars(self, text: str) -> bool:
        """Check if text contains malformed Unicode characters - optimized."""
        # Fast path: check for most common indicator first
        if "�" in text:
            return True
        # Check other indicators only if needed
        return any(ind in text for ind in {"\\ufffd", "\ufffd"})

    def decode_with_recovery(
        self,
        all_tokens: list[int],
        previous_good_text: str = "",
        buffer_tokens: list[int] | None = None,
    ) -> tuple[str, list[int], bool]:
        """Decode tokens with smart error recovery - optimized version."""
        if not all_tokens:
            return "", [], False

        # Combine tokens efficiently
        tokens_to_decode = (buffer_tokens or []) + all_tokens

        # Try full decode first
        try:
            full_decoded = self.processor.decode(tokens_to_decode, skip_special_tokens=True)

            # Fast path: successful clean decode
            if not self.contains_malformed_chars(full_decoded):
                # Extract new text efficiently
                prev_len = len(previous_good_text)
                if prev_len and len(full_decoded) > prev_len and full_decoded[:prev_len] == previous_good_text:
                    return full_decoded[prev_len:], [], False
                return full_decoded, [], False

            # Has malformed chars - use recovery
            return self._recover_partial_decode(tokens_to_decode, previous_good_text, is_error=False)

        except Exception as e:
            logger.debug(f"Decode error: {e}, attempting recovery")
            return self._recover_partial_decode(tokens_to_decode, previous_good_text, is_error=True)

    def _recover_partial_decode(
        self, tokens: list[int], previous_good_text: str, is_error: bool
    ) -> tuple[str, list[int], bool]:
        """Unified recovery logic for both malformed and error cases - optimized with binary search."""
        token_count = len(tokens)

        if token_count <= 1:
            return self.fallback_char, [], True

        # Check cache for this token sequence
        cache_key = (tuple(tokens[: min(8, token_count)]), bool(previous_good_text))
        if cache_key in self._decode_cache:
            self._cache_hits += 1
            cached_point = self._decode_cache[cache_key]
            if cached_point < token_count:
                # Try cached decode point first
                try:
                    partial_decoded = self.processor.decode(tokens[:cached_point], skip_special_tokens=True)
                    if not self.contains_malformed_chars(partial_decoded):
                        new_text = self._extract_new_text(partial_decoded, previous_good_text)
                        return new_text, tokens[cached_point:], True
                except Exception:
                    pass
        else:
            self._cache_misses += 1

        # Binary search for valid decode point (more efficient than linear scan)
        left, right = 1, token_count - 1
        last_good_point = 0
        last_good_text = ""

        while left <= right:
            # Bias towards larger chunks (they're more likely to decode properly)
            mid = (left + right + 1) // 2

            try:
                partial_decoded = self.processor.decode(tokens[:mid], skip_special_tokens=True)

                # Check if this decode point is valid
                if not self.contains_malformed_chars(partial_decoded):
                    last_good_point = mid
                    last_good_text = partial_decoded
                    # Try to decode more tokens
                    left = mid + 1
                else:
                    # This point has malformed chars, try fewer tokens
                    right = mid - 1

            except Exception:
                # Decode failed, try fewer tokens
                right = mid - 1

        # If we found a good decode point, use it
        if last_good_point > 0:
            # Update cache with successful decode point
            if len(self._decode_cache) >= self._cache_size:
                # Simple cache eviction - remove oldest entry
                self._decode_cache.pop(next(iter(self._decode_cache)))
            self._decode_cache[cache_key] = last_good_point

            new_text = self._extract_new_text(last_good_text, previous_good_text)
            remaining_tokens = tokens[last_good_point:]

            if not is_error:
                logger.debug(f"Buffering {len(remaining_tokens)} tokens due to malformed chars")
            else:
                logger.debug(f"Decode error recovery: buffering {len(remaining_tokens)} tokens")

            return new_text, remaining_tokens, True

        # Fall back to linear search from the end if binary search fails
        # (This handles edge cases where binary search might miss valid points)
        for i in range(token_count - 1, 0, -1):
            try:
                partial_decoded = self.processor.decode(tokens[:i], skip_special_tokens=True)
                if not self.contains_malformed_chars(partial_decoded):
                    new_text = self._extract_new_text(partial_decoded, previous_good_text)
                    remaining_tokens = tokens[i:]
                    logger.debug(f"Fallback recovery: buffering {len(remaining_tokens)} tokens")
                    return new_text, remaining_tokens, True
            except Exception:
                continue

        # Complete failure
        logger.warning("Could not find any valid decode point, using fallback")
        return self.fallback_char, [], True

    def _extract_new_text(self, decoded_text: str, previous_text: str) -> str:
        """Extract new text from decoded result - optimized."""
        if not previous_text:
            return decoded_text

        prev_len = len(previous_text)
        # Fast comparison without creating substring first
        if len(decoded_text) > prev_len and decoded_text[:prev_len] == previous_text:
            return decoded_text[prev_len:]
        return decoded_text

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0,
            "cache_size": len(self._decode_cache),
        }
