from easydel.utils import get_logger

logger = get_logger("Decoders")


class SmartBytecodeDecoder:
    """A smart decoder that handles partial token sequences and recovers from malformed characters."""

    def __init__(self, processor, fallback_char: str = ""):
        self.processor = processor
        self.fallback_char = fallback_char
        self.malformed_indicators = {"�", "\\ufffd", "\ufffd"}  # noqa

    def contains_malformed_chars(self, text: str) -> bool:
        """Check if text contains malformed Unicode characters."""
        return any(indicator in text for indicator in self.malformed_indicators)

    def decode_with_recovery(
        self,
        all_tokens: list[int],
        previous_good_text: str = "",
        buffer_tokens: list[int] | None = None,
    ) -> tuple[str, list[int], bool]:
        """Decode tokens with smart error recovery."""
        if not all_tokens:
            return "", [], False
        if buffer_tokens:
            tokens_to_decode = buffer_tokens + all_tokens
        else:
            tokens_to_decode = all_tokens
        try:
            full_decoded = self.processor.decode(tokens_to_decode, skip_special_tokens=True)
            if self.contains_malformed_chars(full_decoded):
                logger.debug("Malformed characters detected in full decode")
                return self._handle_malformed_decode(tokens_to_decode, previous_good_text)
            else:
                if previous_good_text and full_decoded.startswith(previous_good_text):
                    new_text = full_decoded[len(previous_good_text) :]
                else:
                    new_text = full_decoded
                return new_text, [], False

        except Exception as e:
            logger.debug(f"Decode error: {e}, attempting recovery")
            return self._handle_decode_error(tokens_to_decode, previous_good_text)

    def _handle_malformed_decode(self, tokens: list[int], previous_good_text: str) -> tuple[str, list[int], bool]:
        """Handle cases where decoding produces malformed characters."""
        if len(tokens) <= 1:
            return self.fallback_char, [], True
        for i in range(len(tokens) - 1, 0, -1):
            try:
                partial_decoded = self.processor.decode(tokens[:i], skip_special_tokens=True)
                if not self.contains_malformed_chars(partial_decoded):
                    if previous_good_text and partial_decoded.startswith(previous_good_text):
                        good_new_text = partial_decoded[len(previous_good_text) :]
                    else:
                        good_new_text = partial_decoded
                    remaining_tokens = tokens[i:]
                    logger.debug(f"Buffering {len(remaining_tokens)} tokens due to malformed chars")
                    return good_new_text, remaining_tokens, True

            except Exception:
                continue

        logger.warning("Could not find any good partial decode, using fallback")
        return self.fallback_char, [], True

    def _handle_decode_error(self, tokens: list[int], previous_good_text: str) -> tuple[str, list[int], bool]:
        """Handle decode exceptions by trying progressive decoding."""
        if len(tokens) <= 1:
            return self.fallback_char, [], True

        for i in range(len(tokens) - 1, 0, -1):
            try:
                partial_decoded = self.processor.decode(tokens[:i], skip_special_tokens=True)
                if previous_good_text and partial_decoded.startswith(previous_good_text):
                    good_new_text = partial_decoded[len(previous_good_text) :]
                else:
                    good_new_text = partial_decoded
                remaining_tokens = tokens[i:]
                logger.debug(f"Decode error recovery: buffering {len(remaining_tokens)} tokens")
                return good_new_text, remaining_tokens, True

            except Exception:
                continue

        logger.warning("Complete decode failure, using fallback")
        return self.fallback_char, [], True
