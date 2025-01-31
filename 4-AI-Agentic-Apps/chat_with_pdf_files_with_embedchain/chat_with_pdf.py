import streamlit as st
from embedchain import App


app = App()


def main():
    st.title("Chat with Your PDF")
    st.write("Upload a PDF file and ask questions about its content.")

    # File uploader
    pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

    if pdf_file is not None:
        # Save the uploaded PDF to a temporary location
        with open("uploaded_file.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())

        # Add the PDF to the EmbedChain app
        app.add("uploaded_file.pdf", data_type="pdf_file")

        st.success("PDF content has been embedded successfully!")

        # Input for user queries
        query = st.text_input("Ask a question about the PDF:")

        if query:
            # Retrieve the answer from the PDF content
            response = app.query(query)
            st.write("Answer:", response)

if __name__ == "__main__":
    main()