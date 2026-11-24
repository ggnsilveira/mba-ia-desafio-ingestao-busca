from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Faça sua pergunta:\n")

    while True:
        question = input("PERGUNTA: ")
        answer = chain.invoke(question)
        print(f"RESPOSTA: {answer}\n")


if __name__ == "__main__":
    main()
