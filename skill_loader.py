"""Load the Agent Skills standard Markdown instruction file."""
from pathlib import Path

SKILL_PATH = Path(__file__).resolve().parent / 'skills' / 'data-quality-investigator' / 'SKILL.md'


def load_skill() -> str:
    skill = SKILL_PATH.read_text(encoding='utf-8')
    if not skill.startswith('---\n') or '\n---\n' not in skill[4:]:
        raise ValueError('SKILL.md must include YAML frontmatter')
    return skill
