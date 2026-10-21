def truncate_output(text: str, limit: int, source: str = "tool") -> str:
    """
    Truncates text to the specified limit and appends a truncation message.
    
    Args:
        text: The text to truncate.
        limit: The maximum number of characters allowed, including the footer.
        source: The source of the text (e.g., 'shell', 'file', 'search').
        
    Returns:
        The truncated text with a footer if truncation occurred.
    """
    if not text:
        return text
        
    limit = max(0, limit)
    if len(text) <= limit:
        return text

    marker = f"\n... [TRUNCATED from {len(text)} chars | source={source}]"
    if len(marker) > limit:
        return text[:limit]
    return text[:limit - len(marker)] + marker
