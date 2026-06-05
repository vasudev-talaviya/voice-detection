from helper.register import register_voice
from model.machine_learning.predict import predict_voice


def main():
    print("\n===========================================")
    print("   Antigravity Audio Intelligence System   ")
    print("===========================================")
    print("  1. Register new voice")
    print("  2. Predict speaker")
    print("===========================================")

    choice = input("\nPress 1 or 2 : ").strip()

    if choice == "1":
        register_voice()
    elif choice == "2":
        predict_voice()
    else:
        print("[Error] Invalid choice. Please press 1 or 2.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Info] Interrupted by user. Exiting.")
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred: {e}")
