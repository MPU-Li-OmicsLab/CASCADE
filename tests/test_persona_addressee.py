"""人设示例句的说话对象必须是陪伴 AI，不能是家人。

2026-09-11 发现 zh_p08「丧偶独居老人」的示例句全是对着子女说的口吻（「告诉你弟了」
「妈跟你说句实话」「你们别难过」），模拟器照搬 → 全语料 56 句患者发言把 AI 当成了儿子，
180 段金标包里 10 段前文带这种错位。陪伴场景下对话对象是 AI，不是亲属。
"""
import re
import pytest
from cascade.material import load_personas

KIN_ADDRESS = re.compile(r"你弟|你妹|你哥|你姐|你爸|你妈|妈跟你|爸跟你|你们别|麻烦你们")


@pytest.mark.parametrize("persona", [p for p in load_personas().personas if p.language.value == "zh"],
                         ids=lambda p: p.id)
def test_zh_persona_examples_do_not_address_a_family_member(persona):
    for line in list(persona.implicit_examples) + list(persona.explicit_examples):
        assert not KIN_ADDRESS.search(line), f"{persona.id}: 示例句把 AI 当亲属 → {line!r}"
