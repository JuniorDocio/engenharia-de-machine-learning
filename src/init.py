from kedro.framework.startup import register_hook
import streamlit as st

def register_streamlit_server():
    """Função para iniciar o dashboard Streamlit."""
    from kedro.framework.hooks import hook_impl
    from streamlit.web.server import Server as StreamlitServer

    class Server(StreamlitServer):
        def run(self):
            st.stop()
            return super(Server, self).run()

    @hook_impl
    def before_server_start(server):
        if isinstance(server, Server):
            dashboard = __import__("dashboard.app")
            dashboard_session = dashboard.session
            dashboard_session.load_context()
            server.app_mode = True
            server.app = dashboard.app
            server.run_on_save = False

register_hook("before_server_start", "register_streamlit_server", register_streamlit_server)