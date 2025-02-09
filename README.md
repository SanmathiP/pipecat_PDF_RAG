Pipecat is an open-source Python framework designed for building voice-enabled, real-time, multimodal AI applications. 
It manages the complex orchestration of AI services, network transport, audio processing, and multimodal interactions, allowing developers to focus on creating engaging user experiences. 


In the code, Pipecat's MarkdownTextFilter is used, it is a utility that processes text by removing Markdown formatting while preserving its structure and readability.  
By configuring parameters such as enable_text_filter, filter_code, and filter_tables, you can control the filtering process to meet specific needs. 


In the application, the MarkdownTextFilter is initialized with parameters to enable text filtering and remove code blocks and tables. 
This setup ensures that when processing the content of your PDF, any Markdown-specific formatting is stripped away, resulting in clean, plain text that's suitable for further processing or display.

The pdf_rag_chat.py is a plain console interacrive function that takes a PDF as an input and takes a question and answers according to the PDF's content.
The output is as follows:
![image](https://github.com/user-attachments/assets/38152516-7eb7-4c07-9e45-9299c598ae2e)


