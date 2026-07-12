import traceback

print("Testing imports...")

try:
    from app.rag_chat import show_rag_chat
    print("✅ rag_chat imported")
except Exception:
    print("❌ rag_chat failed")
    traceback.print_exc()

try:
    from app.grok_chat import show_grok_chat
    print("✅ grok_chat imported")
except Exception:
    print("❌ grok_chat failed")
    traceback.print_exc()

print("Done")