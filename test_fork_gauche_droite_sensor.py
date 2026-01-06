
print("=== Test Combiné Fourche Gauche ↔ Droite (Capteur réel CENTRE utilisé pour la droite) ===")

from time import sleep
import fork

def dump_left_and_return_center():
    print("\n➡️ Dump à gauche avec capteur")
    err = fork.dumpLeft()
    if err != fork.ERROR_NONE:
        print("❌ Erreur lors du dump à gauche")
        return
    print("⏳ Pause 0.5s avant retour au centre")
    sleep(0.5)

    print("⬅️ Retour au centre (tempo)")
    fork._backward(0.18)
    sleep(1.37)
    fork.stop()
    print("✅ Centre estimé depuis la gauche")

def dump_right_and_return_center():
    print("\n➡️ Dump à droite avec capteur (utilise capteur CENTRE)")
    fork._backward(0.18)
    timeout = 0
    while not fork.switchForkCenter.is_pressed and timeout < 3:
        sleep(0.01)
        timeout += 0.01
    fork.stop()

    if timeout >= 3:
        print("⚠️ Capteur centre non déclenché en allant à droite")
    else:
        print("✅ Capteur centre déclenché (position droite atteinte)")

    print("⏳ Pause 0.5s avant retour au centre")
    sleep(0.5)

    print("⬅️ Retour au centre (tempo)")
    fork._forward(0.18)
    sleep(1.35)
    fork.stop()
    print("✅ Centre estimé depuis la droite")

def test_loop():
    for i in range(3):
        print(f"\n🌀 Cycle {i+1}/3 : GAUCHE")
        dump_left_and_return_center()
        sleep(2)
        print(f"\n🌀 Cycle {i+1}/3 : DROITE")
        dump_right_and_return_center()
        sleep(2)

if __name__ == "__main__":
    try:
        test_loop()
    except KeyboardInterrupt:
        print("Arrêt manuel")
    finally:
        fork.stop()
        fork.pca.deinit()
        print("✅ Test terminé et nettoyage effectué")
