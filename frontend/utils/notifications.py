# frontend/utils/notifications.py
import streamlit as st
import streamlit.components.v1 as components
import time


def show_notification(msg, notification_type="success", refresh=False):
    """Set a notification message to be displayed
    
    Args:
        msg: The message to display
        notification_type: Type of notification - "success", "warning", "error", "info"
    """
    # Initialize notification state if not present
    if "notification" not in st.session_state:
        st.session_state.notification = None
    if "notification_time" not in st.session_state:
        st.session_state.notification_time = 0
    if "notification_type" not in st.session_state:
        st.session_state.notification_type = "success"

    st.session_state.notification = msg
    st.session_state.notification_type = notification_type
    st.session_state.notification_time = time.time()
    display_floating_notification()
    if refresh:
        time.sleep(3)
        st.rerun()


def display_floating_notification():
    """Display a floating notification if one is set"""
    if st.session_state.notification:
        time_elapsed = time.time() - st.session_state.notification_time
        
        # Get notification type and corresponding color
        notification_type = st.session_state.get("notification_type", "success")
        
        # Define colors for different notification types
        colors = {
            "success": "#4CAF50",  # Green
            "warning": "#FF9800",  # Orange/Yellow
            "error": "#F44336",    # Red
            "info": "#2196F3"      # Blue
        }
        
        bg_color = colors.get(notification_type, colors["success"])
        
        if time_elapsed < 3:
            # Show floating notification
            components.html(
                f"""
                <div id="notification" style="
                    position: fixed;
                    top: 0; left: 0; width: 94%;
                    background-color: {bg_color}; color: white; padding: 10px 20px;
                    border-radius: 5px; z-index: 9999; box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                    font-weight: 500; text-align: center;
                ">
                    {st.session_state.notification}
                </div>

                <script>
                    setTimeout(function() {{
                        var n = document.getElementById("notification");
                        if (n) {{
                            n.style.display = 'none';
                        }}
                    }}, {3000});
                </script>
                """,
                height=80
            )
        else:
            # Auto-hide after 3 seconds
            st.session_state.notification = None
            st.session_state.notification_type = "success"
            st.session_state.notification_time = 0
