const path = require('path')
require('dotenv').config({ path: path.join(__dirname, '.env')  })
const express = require('express')
const morgan = require('morgan')

const PORT = process.env.PORT
const app = express()

app.use(morgan('common'))
app.use(express.static(path.join(__dirname, '..', 'frontend')))

app.listen(PORT)
