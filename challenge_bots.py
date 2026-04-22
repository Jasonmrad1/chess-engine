import os
import berserk
import time

# Load token from environment
TOKEN = os.getenv("LICHESS_TOKEN")

if not TOKEN:
    raise RuntimeError("Missing LICHESS_TOKEN environment variable")

session = berserk.TokenSession(TOKEN)
client = berserk.Client(session=session)

# ✅ REAL bots (NOT stockfish_level_X)
TARGET_BOTS = [
    # "Lynx_BOT",            # ~2700
    # "likeawizard-bot"            # ~2100
    # "maia9"                    # ~2000
    # "halcyonbot"
    # "StrawberryChessDev"
    "Boosted_Maia_1700"
    ]

def challenge(bot_id):
    print(f"Challenging {bot_id}...")

    try:
        client.challenges.create(
            bot_id,
            rated=False,          # ❗ IMPORTANT: bots often reject rated games
            clock_limit=300,      # 5 minutes
            clock_increment=3,
            color="random",
            variant="standard"
        )
        print(f"Challenge sent to {bot_id}")

    except Exception as e:
        print(f"❌ Failed vs {bot_id}: {e}")

while True:
    for bot in TARGET_BOTS:
        challenge(bot)
        time.sleep(15)  # ✅ avoid rate limit

    print("🔁 Cycle done, waiting 60 seconds...\n")
    time.sleep(60)