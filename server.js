const express = require("express");
const app = express();

app.get("/", (req, res)=>{
    console.log(req.query);
    return res.send("<h1>Bienvenido</h1>");
});

app.listen(8080, function(){
    console.log("The server is running");
});