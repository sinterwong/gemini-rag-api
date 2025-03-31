import uuid
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class Document:
    """表示包含元数据的文档"""

    # 文档文本内容
    text: str

    # 文档元数据（可选）
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 文档唯一ID（如果不提供则自动生成）
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于序列化"""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """从字典创建文档"""
        return cls(
            text=data["text"],
            metadata=data.get("metadata", {}),
            doc_id=data["doc_id"]
        )
