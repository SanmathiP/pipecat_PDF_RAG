Pipecat is an open-source Python framework designed for building voice-enabled, real-time, multimodal AI applications. 
It manages the complex orchestration of AI services, network transport, audio processing, and multimodal interactions, allowing developers to focus on creating engaging user experiences. 


In the code, Pipecat's MarkdownTextFilter is used, it is a utility that processes text by removing Markdown formatting while preserving its structure and readability.  
By configuring parameters such as enable_text_filter, filter_code, and filter_tables, you can control the filtering process to meet specific needs. 


In the application, the MarkdownTextFilter is initialized with parameters to enable text filtering and remove code blocks and tables. 
This setup ensures that when processing the content of your PDF, any Markdown-specific formatting is stripped away, resulting in clean, plain text that's suitable for further processing or display.

The pdf_rag_chat.py is a plain console interacrive function that takes a PDF as an input and takes a question and answers according to the PDF's content.
The output is as follows:

![image](https://github.com/user-attachments/assets/38152516-7eb7-4c07-9e45-9299c598ae2e)





The pipecat_new folder has 2 python files pipecat_pdf_rag.py and pipecat_pdf_rag_Storytelling.py
These involve pipecat.utils function to make use of MarkdownFilter. 

![image](https://github.com/user-attachments/assets/f4521da3-4947-486f-964e-f20bd95a4bac)





The stortelling chat answers to the questions in a bright Bavarian manner to make the answers more interesting!, as show below:
![image](https://github.com/user-attachments/assets/a2fc5a8c-4a66-4623-a907-d51ed021c5b5)




The usecase can be further taken to the next level by adding voice, vision and a better interface in the future!



