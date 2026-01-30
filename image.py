<html>
<body>
    <script src="https://js.puter.com/v2/"></script>
    <img src="https://images.news18.com/ibnlive/uploads/2026/01/Virat-Kohli-2026-01-481622ac8e8d300e8dca26bff60a7640-16x9.jpg" style="display:block;">
    <script>
        // Loading ...
        puter.print(`Loading...`);

        // Image analysis with GPT-5 nano
        puter.ai
            .chat(`Give opensource inteligence report in point wise ?`, `https://images.news18.com/ibnlive/uploads/2026/01/Virat-Kohli-2026-01-481622ac8e8d300e8dca26bff60a7640-16x9.jpg`, {
                model: "gpt-5-nano",
            })
            .then(puter.print);
    </script>
</body>
</html>