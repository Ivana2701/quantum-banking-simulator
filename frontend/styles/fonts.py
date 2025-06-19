import streamlit as st

def load_particle_background():
    st.markdown("""
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r121/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/vanta@latest/dist/vanta.net.min.js"></script>
        <div id="vanta-bg" style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; z-index: -1;"></div>
        <script>
        VANTA.NET({
          el: "#vanta-bg",
          mouseControls: true,
          touchControls: true,
          minHeight: 200.00,
          minWidth: 200.00,
          scale: 1.0,
          scaleMobile: 1.0,
          color: 0x0f0eff,
          backgroundColor: 0x0a0a0a
        })
        </script>
    """, unsafe_allow_html=True)

def load_matrix_background():
    st.markdown("""
        <style>
        canvas#matrix {
            position: fixed;
            z-index: -1;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: black;
        }
        </style>
        <canvas id="matrix"></canvas>
        <script>
        var c = document.getElementById("matrix");
        var ctx = c.getContext("2d");
        c.height = window.innerHeight;
        c.width = window.innerWidth;
        var matrix = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ123456789@#$%^&*()*&^%+-/~{[|`]}";
        matrix = matrix.split("");
        var font_size = 10;
        var columns = c.width / font_size;
        var drops = [];
        for (var x = 0; x < columns; x++) drops[x] = 1;
        function draw() {
            ctx.fillStyle = "rgba(0, 0, 0, 0.05)";
            ctx.fillRect(0, 0, c.width, c.height);
            ctx.fillStyle = "#0f0"; 
            ctx.font = font_size + "px monospace";
            for (var i = 0; i < drops.length; i++) {
                var text = matrix[Math.floor(Math.random() * matrix.length)];
                ctx.fillText(text, i * font_size, drops[i] * font_size);
                if (drops[i] * font_size > c.height && Math.random() > 0.975) drops[i] = 0;
                drops[i]++;
            }
        }
        setInterval(draw, 33);
        </script>
    """, unsafe_allow_html=True)
