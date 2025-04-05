from typing import List, Dict, Any
from models.document import Document

import requests
import json
import re


def parse_my_weird_txt(file_path: str, delimiter: str = '\n\n\n\n\n\n') -> List[Document]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            raw_docs = content.split(delimiter)
    except FileNotFoundError:
        print(f"文件 {file_path} 没找到！")
        return []
    except Exception as e:
        print(f"读文件 {file_path} 出错: {e}")
        return []

    parsed_documents = []

    for i, doc_part in enumerate(raw_docs):
        if not doc_part.strip():
            continue

        lines = doc_part.strip().split('\n')
        if i == 0 and not content.startswith('# '):
            pass
        elif i > 0:
            pass

        title = ""
        metadata = {}
        text_lines = []
        in_text_section = False

        current_lines = doc_part.strip().split('\n')
        if not current_lines:
            continue

        title = current_lines[0].strip()
        metadata['title'] = title.replace("# ", "")

        line_idx = 1
        while line_idx < len(current_lines):
            line = current_lines[line_idx].strip()
            if line.startswith('- '):
                match = re.match(r'-\s*([^:]+):\s*(.*)', line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    metadata[key] = value
            elif line.startswith('## 正文'):
                in_text_section = True
                line_idx += 1
                break
            elif not line:
                pass
            else:
                print(f"Warning: 在文档 '{title}' 中发现非预期的行: {line}")
                pass
            line_idx += 1

        if in_text_section:
            text_content = "\n".join(current_lines[line_idx:]).strip()
        else:
            print(f"Warning: 文档 '{title}' 未找到 '## 正文' 标记，将剩余部分视为正文。")
            text_content = "\n".join(current_lines[line_idx:]).strip()  # 简化处理

        if text_content:
            parsed_document = {
                "doc_id": metadata["来源ID"],
                "text": text_content,
                "metadata": metadata
            }
            doc = Document.from_dict(parsed_document)
            parsed_documents.append(doc)
        else:
            print(f"警告: 文档 '{title}' 解析后没有正文内容。")

    return parsed_documents


def add_documents_to_api(ip: str, port: int, documents: List[Dict[str, Any]]):
    url = f"http://{ip}:{port}/documents"
    headers = {'Content-Type': 'application/json'}
    payload = {'documents': documents}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # 抛出异常如果状态码不是2xx
        print(f"成功添加文档到API: {response.json()}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP 错误: {http_err}")
        if response.text:
            print(f"服务器返回内容: {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"连接错误: {conn_err}")
    except Exception as err:
        print(f"其他错误: {err}")


def query_api(ip: str, port: int, query_text: str, top_k: int = 5):
    url = f"http://{ip}:{port}/query"
    headers = {'Content-Type': 'application/json'}
    payload = {'query_text': query_text, 'top_k': top_k}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"查询结果: {response.json()}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP 错误: {http_err}")
        if response.text:
            print(f"服务器返回内容: {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"连接错误: {conn_err}")
    except Exception as err:
        print(f"其他错误: {err}")


def generate_api(ip: str, port: int, query_text: str, top_k: int = 5, include_sources: bool = True):
    url = f"http://{ip}:{port}/generate"
    headers = {'Content-Type': 'application/json'}
    payload = {'query_text': query_text, 'top_k': top_k,
               'include_sources': include_sources}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"生成结果: {response.json()}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP 错误: {http_err}")
        if response.text:
            print(f"服务器返回内容: {response.text}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"连接错误: {conn_err}")
    except Exception as err:
        print(f"其他错误: {err}")


if __name__ == "__main__":
    ip = "localhost"
    port = 9797

    file_path = "data/documents/init_1.txt"
    parsed_docs = parse_my_weird_txt(file_path)

    if parsed_docs:
        print(f"成功解析了 {len(parsed_docs)} 个文档:")

    documents_for_api = [
        {"text": doc.text, "metadata": doc.metadata} for doc in parsed_docs[:3]
    ]
    add_documents_to_api(ip, port, documents_for_api)

    query_api(ip, port, "西汉姆球员最近有什么动态？")

    generate_api(ip, port, "西汉姆球员最近有什么动态？")
