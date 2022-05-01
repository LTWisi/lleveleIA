from flask import Flask, request


from objectDefiner import objectDefiner
classInstance = objectDefiner()

app = Flask(__name__)

@app.route('/analisys', methods=['POST'])
def analisys():
    data = request.get_json(force=True)
    url = data['url']

    resultado = classInstance.getImageCategory(url)
    response = {
        'msg' : 'ANALISYS OK',
        'docs' : {
            'url' : url,
            'categoria' : resultado
        },
        'controller' : 'test',
        'code' : 200
    }
    return response, 200, {'ContentType':'application/json'} 

@app.route('/test', methods=['GET'])
def text():
    response = {
        'msg' : 'QUERY OK',
        'docs' : {'null' : 'null'},
        'controller' : 'test',
        'code' : 200
    }
    return response, 200, {'ContentType':'application/json'} 

@app.route('/', methods=['GET'])
def rootf():
    return '<h1> Llévele Servidor Secundario.</h1>'

if __name__ == "__main__":
    app.run(port=5000, debug=True)

