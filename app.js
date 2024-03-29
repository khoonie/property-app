const express = require("express");
const bodyParser = require("body-parser");
const expressHbs = require("express-handlebars");

const props = require("./data/web_prop_data.json")
const model = require("./model")

const app = express();
app.set("views", "./views");
app.set("view engine", "hbs");

//Body parser middleware
app.use(
    bodyParser.urlencoded({
        extended: false
    })
);
app.use(bodyParser.json());


app.engine('.hbs', expressHbs({
    defaultLayout: 'layout',
    extname: '.hbs'
}));


app.get("/", (req, res) => {    res.render("index", { props: props.slice(0, 12), pg_start: 0, pg_end: 12 })});

app.get("/get-next", (req, res) => {
    let pg_start = Number(req.query.pg_end)
    let pg_end = Number(pg_start) + 12
    console.log("next page")
    res.render("index", {
        props: props.slice(pg_start, pg_end),
        pg_start: pg_start,
        pg_end: pg_end
    })
});


app.get("/get-prev", (req, res) => {
    let pg_end = Number(req.query.pg_start)
    let pg_start = Number(pg_end) - 12

    if (pg_start <= 0) {
        res.render("index", { props: props.slice(0, 12), pg_start: 0, pg_end: 12 })

    } else {
        res.render("index", {
            props: props.slice(pg_start, pg_end),
            pg_start: pg_start,
            pg_end: pg_end
        })

    }
});

app.get("/recommend", (req, res) => {
    let userId = req.query.userId
    if (Number(userId) > 2000 || Number(userId) < 0) {
        res.send("User Id cannot be greater than 2000 or less than 0!")
    } else {
        recs = model.recommend(userId)
            .then((recs) => {
                res.render("index", { recommendations: recs, forUser: true })
            })
    }

})

module.exports = app;