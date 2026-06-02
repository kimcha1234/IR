from pathlib import Path


def test_readme_documents_chatbot_gui_demo():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Chatbot GUI Demo" in readme
    assert "streamlit run robot_drawing_planner/demo_app.py" in readme
    assert "one message per tool call" in readme
