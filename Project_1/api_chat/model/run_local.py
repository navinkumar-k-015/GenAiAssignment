import os,sys

import uvicorn


if __name__ == "__main__":
    from predictor import app
    
    uvicorn.run("predictor:app", port=8080,reload=True,workers=1)
