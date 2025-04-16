
<div align="center">

# TOOGLE: Text Summarizer 📑 
➳ [TOOGLE Documentation](https://github.com/bubblebolt/dads/blob/main/DADS5001/ASM5-LLM/Toogle.pdf)
&nbsp;&nbsp;&nbsp;
➳ [TOOGLE Presentation](https://www.canva.com/design/DAF0xpmxLW8/9MnZXA5cfAM1KEzETG6skA/view?utm_content=DAF0xpmxLW8&utm_campaign=designshare&utm_medium=link&utm_source=editor)
&nbsp;&nbsp;&nbsp;
➳ [Download TOOGLE App](https://github.com/bubblebolt/dads/raw/main/DADS5001/ASM5-LLM/Toogle_Readytoload.zip)


![Alt Text](https://github.com/bubblebolt/dads/raw/main/DADS5001/ASM5-LLM/Pics/ex_toogle.gif)


</div>


## Introduction 💬
TOOGLE is a text summarization tool designed to streamline data analysis by providing concise and reliable summaries from large volumes of text or documents. This README.md serves as a guide to understand the project's origin, objectives, methodology, expected results, limitations, and recommendations.

## Origin and Significance 🔎
TOOGLE emphasizes the importance of data and data management in contemporary contexts, where data plays a pivotal role in business decision-making and socio-physical analysis. The tool aims to simplify data analysis in scenarios with vast and complex datasets, enabling analysts to efficiently focus on critical information.

### Processing Data Models 
- 'facebook/bart-large-cnn': Used for text summarization.
- 'facebook/mbart-large-50-many-to-many-mmt': Used for language translation.
- 'cardiffnlp/tweet-topic-21-multi': Used for text categorization.

## Objectives 🎯
1. Understand and utilize Dash, Plotly, and Python for creating visually appealing graphs and data representation.
2. Develop a tool capable of generating concise and reliable content summaries, aiming to reduce complexity and increase efficiency in utilizing crucial data effectively.

## Methodology 📦
- **Data Collection:** Users input text via a Textarea on the Dash website, selecting the text language from a dropdown menu. The system automatically identifies and verifies the language using the Langid module.
- **Data Storage:** Processed data from text analysis is stored in a PostgreSQL database, including ID, Timestamp, Topic, and Probability.
<div align="center">
<img src="https://raw.githubusercontent.com/bubblebolt/dads/main/DADS5001/ASM5-LLM/Pics/db.png" width="500"> 
</div>

## Expected Results and Data Presentation 📊
TOOGLE is expected to produce curated and significant data summaries, allowing analysts to quickly focus on essential information. The presentation includes displaying Topic & Probabilities in graphical formats, such as horizontal bar charts, for easy comprehension using Plotly.

## Limitations and Recommendations 🔒
- **Language Support:** The 'facebook/bart-large-cnn' model supports summarization only in English. Consider models supporting multiple languages.
- **Translation Limitations:** The 'facebook/mbart-large-50-many-to-many-mmt' model has limitations in translating between certain languages. Explore models with broader capabilities.
- **Summary Length Constraints:** Users are constrained to select summary lengths ranging from 20-200 characters, potentially leading to loss of important information. Enhance flexibility in adjusting summary lengths for better user customization and content coverage.



## Collaborators 🤝🏻

| Name          | Student ID  |
|---------------|-------------|
| Aritsara Intasaeng | 6610412001 |
| Chalita Iamleelaporn | 6610412002 |
| Phornpailin Thaisuriyo | 6610412006 |
| Monsicha Praditrod | 6610412007 |

For further details and contributions, refer to the project repository.
