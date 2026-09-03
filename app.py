import os
import re
import time
import joblib

def limpiar_texto(texto):
    texto = str(texto).lower()
    return re.sub(r'[^\w\s]', '', texto)

print("\nIniciando Asistente Virtual Judicial...")
clasificador = joblib.load(os.path.join('modelos_exportados', 'clasificador_lr.pkl'))
vectorizador = joblib.load(os.path.join('modelos_exportados', 'vectorizador_tfidf.pkl'))

print("¡Asistente listo!\n")
print("=" * 50)

while True:
    print("\n⚖️  Asistente Judicial - Atención Ciudadana")
    print("Hola, soy tu asistente virtual. ¿En qué te ayudo hoy?")
    print("1️⃣  Consultar Sedes y Horarios")
    print("2️⃣  Ver tarifario de Aranceles")
    print("3️⃣  Reportar una emergencia legal")
    print("4️⃣  Salir del chat")
    
    opcion = input("\n👤 Tu respuesta (1-4): ")
    
    # FLUJO 1: RESPUESTAS DIRECTAS
    if opcion == '1':
        print("\n🤖 Asistente: Aquí tienes los horarios de nuestras sedes principales:")
        print("   📍 Sede Central (Cercado): Lunes a Viernes, 08:00 hrs - 16:45 hrs.")
        print("   📍 Sede Familia y Penal: Lunes a Viernes, 08:00 hrs - 13:00 hrs (Turno Mañana).")
        print("   ❗ Nota: Mesa de partes virtual opera las 24 horas.")
        input("\n (Presiona ENTER para volver al menú principal...)")
        
    elif opcion == '2':
        print("\n🤖 Asistente: Estos son los costos para trámites comunes:")
        print("   📄 Ofrecimiento de pruebas: S/ 49.50")
        print("   📄 Recurso de Apelación: S/ 198.00")
        print("   📄 Copias Certificadas (por folio): S/ 4.90")
        input("\n(Presiona ENTER para volver al menú principal...)")
        
    elif opcion == '4':
        print("\n🤖 Asistente: Gracias por comunicarte con nosotros. ¡Hasta pronto!")
        break
        
    # FLUJO 3: CLASIFICACION CON NLP
    elif opcion == '3':
        print("\n🤖 Asistente: Entendido. Describe tu situación legal para derivarte al área correcta:")
        mensaje_entrante = input("👤 Describe tu caso: ")
        
        print("🤖 Asistente: Analizando la consulta...")
        time.sleep(1) # Simula un tiempo de procesamiento
        
        mensaje_limpio = limpiar_texto(mensaje_entrante)
        matriz_tfidf = vectorizador.transform([mensaje_limpio])

        # Validacion de vocabulario legal conocido
        if matriz_tfidf.nnz == 0:
            print("\n⚠️ Asistente: No logramos identificar términos legales válidos en tu consulta.")
            solicitar_asesor = input("¿Deseas ser transferido a un asesor? (S/N): ").strip().lower()
            if solicitar_asesor == 's':
                dni = input("👤 Ingresa tu número de DNI para transferirte a la cola de atención: ")
                print(f"✅ Solicitud registrada para el DNI {dni}. Un asesor te atenderá en breve.")
            else:
                print("ℹ️ Entendido. Regresando al menú principal para que puedas reformular tu consulta.")
        else:
            urgencia = clasificador.predict(matriz_tfidf)[0]
        
            if urgencia == "Alta":
                print("\n🚨 Asistente (ALERTA): Caso clasificado con prioridad ALTA.")
                dni = input("👤 Por favor, ingresa tu número de DNI para alertar al juzgado de turno: ")
                print(f"✅ Alerta con prioridad ALTA generada para el DNI {dni}. Un especialista te contactará a la brevedad.")
            
            elif urgencia == "Media":
                print("\n⚖️ Asistente: Caso clasificado con prioridad MEDIA.")
                expediente = input("👤 ¿Cuentas con un número de expediente? (Escribe el número o 'No'): ")
                if expediente.lower() == 'no':
                    print("✅ Solicitud derivada a mesa de partes. Respuesta estimada en 24 horas.")
                else:
                    print(f"✅ Consulta vinculada al expediente {expediente}. Se notificará al especialista asignado.")
                
            else:
                print("\nℹ️ Asistente: Caso clasificado con prioridad BAJA).")
                print("✅ Ticket de atención generado. Un asesor te responderá por este canal durante el día.")
        input("\n(Presiona ENTER para volver al menú principal...)")
        
    else:
        print("\n❌ Opción no válida. Por favor, escribe un número del 1 al 4.")