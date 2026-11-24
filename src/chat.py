from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta:\n")

    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not question:
            continue

        answer = chain.invoke(question)
        print(f"RESPOSTA: {answer}\n")


if __name__ == "__main__":
    main()
