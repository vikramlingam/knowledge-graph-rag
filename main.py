"""
The Hive Lab - RAG Application with Knowledge Graph
NiceGUI-based frontend for document Q&A and graph visualization.
Pitch Black Theme Edition.
"""

# CRITICAL STABILITY FIXES FOR MACOS
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import asyncio
import logging
import io
import json
from typing import Optional, List, Dict, Any, Tuple
from nicegui import ui, events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
vector_store: Optional[Tuple[Any, List[dict]]] = None
document_chunks: List[dict] = []
uploaded_file_contents: List[Dict[str, Any]] = []
graph_data_json: str = ""


def process_documents_sync(file_contents: List[Dict]) -> Tuple[Any, List[dict]]:
    """Process documents - runs in thread."""
    from src.ingestion import extract_text
    from src.rag import build_vector_store
    
    all_chunks = []
    for file_data in file_contents:
        class MockFile:
            def __init__(self, name: str, content: bytes):
                self.name = name
                self._buffer = io.BytesIO(content)
            def getvalue(self):
                return self._buffer.getvalue()
            def read(self, *args):
                return self._buffer.read(*args)
            def seek(self, *args):
                return self._buffer.seek(*args)
            def tell(self):
                return self._buffer.tell()
        
        mock_file = MockFile(file_data['name'], file_data['content'])
        chunks = extract_text(mock_file)
        all_chunks.extend(chunks)
        logger.info(f"Extracted {len(chunks)} chunks from {file_data['name']}")
    
    if not all_chunks:
        return (None, []), []
    
    return build_vector_store(all_chunks), all_chunks


def build_graph_sync(chunks: List[dict]) -> str:
    """Build knowledge graph data - runs in thread."""
    from src.knowledge_graph import build_graph_data, get_vis_js_html
    graph_data = build_graph_data(chunks)
    return get_vis_js_html(graph_data)


def prepare_rag_prompt_sync(query: str, vs: Tuple[Any, List[dict]]) -> Tuple[str, str]:
    """Prepare prompt for RAG - runs in thread."""
    from src.rag import retrieve
    
    if vs is None:
        return None, "Please upload and process documents first."
    
    try:
        relevant = retrieve(query, vs)[:5]
        if not relevant:
            return None, "No relevant information found."
        
        context_parts = []
        for chunk in relevant:
            src = f"{chunk['source']} (Page {chunk['page']})"
            text = chunk['text'][:2500] 
            context_parts.append(f"[{src}]: {text}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = f"""You are an assistant. Answer the user's question using the Context below.
        
Context:
{context}

Instructions:
1. Write a clear, detailed answer to the question based on the text above.
2. After your answer, include the Source and Page Number.

Example:
The corporate tax rate is 9%. [Source: Law.pdf (Page 5)]

Question: {query}
Answer:"""


        
        return system_prompt, None
    except Exception as e:
        logger.error(f"Error: {e}")
        return None, f"Error: {str(e)}"


@ui.page('/')
async def main_page():
    """Main application page."""
    global vector_store, uploaded_file_contents, document_chunks, graph_data_json
    
    uploaded_file_contents = []
    document_chunks = []
    graph_data_json = ""
    
    # CSS: Pitch Black + Blue/Cyan Theme
    ui.add_head_html('''
    <style>
        :root {
            --q-primary: #00CCFF; 
            --q-dark: #000000;
        }
        body, html { 
            margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden !important; background-color: #000000; color: #ffffff; 
        }
        .q-page-container { padding: 0 !important; }
        .q-layout { background-color: #000000; }
        
        #kg-container { position: absolute; top: 0; left: 0; right: 0; bottom: 0; width: 100%; height: 100%; background: #000000; }
        
        /* Blue Glass Side Pane */
        .glass-pane {
            background: rgba(10, 20, 30, 0.95);
            backdrop-filter: blur(15px);
            border-left: 1px solid #005577;
            box-shadow: -10px 0 30px rgba(0, 200, 255, 0.2);
        }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    ''')
    
    
    # Layout
    with ui.element('div').classes('fixed top-0 left-0 w-full h-full bg-black overflow-hidden flex'):
        
        # Sidebar
        with ui.element('div').classes('w-72 bg-black border-r border-gray-900 p-4 flex-shrink-0 flex flex-col z-20 relative'):
            ui.label("Knowledge Graph").classes('text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500 mb-6')
            
            ui.label("Upload Documents").classes('text-sm font-semibold text-gray-400 mb-2')
            status = ui.label("No documents selected").classes('text-xs text-gray-600 mb-4')
            
            async def on_upload(e: events.UploadEventArguments):
                global uploaded_file_contents
                content = e.content.read()
                uploaded_file_contents.append({'name': e.name, 'content': content})
                status.set_text(f"{len(uploaded_file_contents)} file(s) ready")
                status.classes('text-green-500')
            
            ui.upload(label='Drop files here', multiple=True, auto_upload=True, on_upload=on_upload).props('accept=".pdf,.docx,.txt" flat bordered color="cyan"').classes('w-full mb-4 bg-gray-900 rounded')
            
            async def process():
                global vector_store, document_chunks, graph_data_json
                if not uploaded_file_contents:
                    status.set_text("Please upload files first")
                    return
                
                status.set_text("Processing...")
                status.classes('text-cyan-400')
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, process_documents_sync, uploaded_file_contents)
                    (idx, chunks), all_chunks = result
                    
                    if idx:
                        vector_store = (idx, chunks)
                        document_chunks = all_chunks
                        status.set_text(f"Indexed {len(chunks)} chunks")
                        status.classes('text-green-500')
                        
                        status.set_text("Building Knowledge Graph...")
                        graph_data_json = await loop.run_in_executor(None, build_graph_sync, all_chunks)
                        status.set_text(f"Ready! {len(chunks)} chunks")
                        
                        tabs.value = graph_tab
                        await asyncio.sleep(0.5)
                        
                        # Init Graph
                        data = json.loads(graph_data_json)
                        await ui.run_javascript(f'''
                            console.log("Init Graph...");
                            var container = document.getElementById('kg-container');
                            if(container && typeof vis !== 'undefined') {{
                                try {{
                                    var nodes = new vis.DataSet({data["nodes"]});
                                    var edges = new vis.DataSet({data["edges"]});
                                    var options = {{
                                        nodes: {{ 
                                            shape: 'dot', 
                                            font: {{ color: '#fff', size: 10, face: 'Inter' }}, 
                                            borderWidth: 1,
                                            shadow: {{ enabled: true, color: 'rgba(0,255,255,0.3)', size: 5 }}
                                        }},
                                        edges: {{ 
                                            smooth: {{ type: 'continuous' }}, 
                                            color: {{ color: 'rgba(0, 204, 255, 0.4)', highlight: '#FFFF00' }},
                                            width: 1,
                                            shadow: false
                                        }},
                                        physics: {{
                                            forceAtlas2Based: {{ 
                                                gravitationalConstant: -20, 
                                                centralGravity: 0.005, 
                                                springLength: 150, 
                                                springConstant: 0.05,
                                                damping: 0.4
                                            }},
                                            solver: 'forceAtlas2Based',
                                            stabilization: {{ iterations: 80 }}
                                        }},
                                        interaction: {{ hover: true, tooltipDelay: 200, zoomView: true }}
                                    }};
                                    var network = new vis.Network(container, {{nodes: nodes, edges: edges}}, options);
                                    
                                    network.on("stabilizationIterationsDone", function () {{
                                        network.fit({{ animation: {{ duration: 1000 }} }});
                                    }});
                                    
                                }} catch(e) {{ console.error(e); }}
                            }}
                        ''', timeout=120.0)
                except TimeoutError:
                    pass 
                except Exception as e:
                    status.set_text(f"Error: {e}")
            
            ui.button("PROCESS DOCUMENTS", on_click=process).classes('w-full bg-cyan-600 text-black font-bold py-2 rounded shadow-lg hover:bg-cyan-500')

        # Main Area
        with ui.element('div').classes('flex-1 flex flex-col bg-black relative'):
            with ui.tabs().classes('w-full text-gray-400 z-20 bg-black border-b border-gray-800') as tabs:
                chat_tab = ui.tab('CHAT')
                graph_tab = ui.tab('KNOWLEDGE GRAPH')
            
            with ui.tab_panels(tabs, value=chat_tab).classes('flex-1 w-full bg-black p-0 relative'):
                
                # CHAT PANE
                with ui.tab_panel(chat_tab).classes('h-full flex flex-col p-6'):
                    # Use a scroll area for messages to ensure proper vertical stacking
                    message_container = ui.scroll_area().classes('flex-1 w-full mb-4 border border-gray-900 rounded-lg bg-gray-900/50')
                    
                    with ui.row().classes('w-full gap-4 items-center bg-gray-900 p-4 rounded-lg border border-gray-800'):
                        inp = ui.input(placeholder='Ask...').classes('flex-1 text-white').props('dark borderless')
                        
                        async def send():
                            from src.local_llm import LocalLLM
                            q = inp.value.strip()
                            if not q: return
                            inp.value = ""
                            
                            # Add User Message to Scroll Area
                            with message_container:
                                ui.html(f'<div style="align-self: flex-end; margin-left: auto; background: #222; padding: 10px; border-radius: 8px; color: white; margin-bottom: 10px; max-width: 80%;">{q}</div>')
                            
                            if not vector_store: return
                            
                            # Add Response Container immediately
                            with message_container:
                                response_html = ui.html('').classes('w-full mb-4')
                                loading = ui.label("Thinking...").classes('text-cyan-400 italic text-sm ml-2')
                            
                            try:
                                loop = asyncio.get_event_loop()
                                sys_prompt, err = await loop.run_in_executor(None, prepare_rag_prompt_sync, q, vector_store)
                                if err: 
                                    loading.delete()
                                    response_html.content = f"<div style='color:red'>{err}</div>"
                                    return
                                
                                llm = LocalLLM()
                                stream = llm.generate_stream(q, sys_prompt)
                                content = ""
                                loading.delete()
                                
                                for chunk in stream:
                                    content += chunk
                                    response_html.content = f'<div style="background: #111; padding: 15px; border-radius: 8px; border-left: 3px solid cyan; color: #ddd; max-width: 80%;">{content.replace(chr(10), "<br>")}</div>'
                                    message_container.scroll_to(percent=1.0) # Auto-scroll to bottom
                                    await asyncio.sleep(0)
                                    
                            except Exception as e:
                                loading.delete()
                                response_html.content = f"<div style='color:red'>Error: {e}</div>"
                        
                        ui.button(icon='send', on_click=send).props('flat round color="cyan"')
                        inp.on('keydown.enter', send)

                # GRAPH PANE
                with ui.tab_panel(graph_tab).classes('h-full w-full p-0 relative overflow-hidden'):
                    ui.html('<div id="kg-container"></div>').classes('w-full h-full')
                    
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Knowledge Graph", port=8080, reload=False, dark=True)
