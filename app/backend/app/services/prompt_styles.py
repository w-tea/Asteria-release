def apply_advanced_style_text(base: str, style_text: str) -> str:
    value = str(style_text or '').strip()
    if not value:
        return base
    if '{prompt}' in value:
        return value.replace('{prompt}', base).strip()
    if not base.strip():
        return value
    return f'{base.rstrip()}, {value}'


def compose_with_advanced_styles(prompt: str, negative_prompt: str, styles: list) -> tuple[str, str]:
    for style in sorted(styles, key=lambda item: item.order):
        prompt = apply_advanced_style_text(prompt, style.prompt)
        negative_prompt = apply_advanced_style_text(negative_prompt, style.negative)
    return prompt, negative_prompt
