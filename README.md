\section*{Local LLM Notebooks and Utilities}

This repository contains a collection of local LLM notebooks and helper scripts. Most models make use of the \texttt{transformers} library provided by Hugging Face and are designed to run locally.

\subsection*{Structure and Usage}

\begin{itemize}
    \item \textbf{Training}: Notebooks (\texttt{.ipynb}) are typically converted to Python scripts (\texttt{.py}) and run in the background using \texttt{tmux}. Logs are saved for later review.
    \item \textbf{Control}: Scripts are included to stop training by terminating the corresponding \texttt{tmux} session.
\end{itemize}

\subsection*{Search-Based Models}

Some models require a running instance of a \texttt{searXNG} container for retrieval-augmented generation (RAG). Ensure this container is correctly set up and running before using search-integrated models.

\subsection*{Environment Note}

These scripts assume you are using a Conda environment named \texttt{llm-env}.  
You will need to manually set up your own environment with the required dependencies. No automatic environment setup is currently provided.

\subsection*{Coming Soon}

\begin{itemize}
    \item Setup instructions
    \item Docker container for full environment and dependency setup
\end{itemize}
