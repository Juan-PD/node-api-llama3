import express from 'express';
import { HfInference } from '@huggingface/inference';

const app = express();
const hfInference = new HfInference("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");

app.use(express.json());

app.post('/api/model', async (req, res) => {
    const { inputText } = req.body;

    try {
        const response = await hfInference.textGeneration({
            model: 'meta-llama/Meta-Llama-3-8B-Instruct',
            inputs: inputText,
            parameters: {
                max_new_tokens: 100,
                temperature: 0.7,
                top_p: 0.9,
                repetition_penalty: 1.2
            }
        });

        res.json(response);
    } catch (error) {
        res.status(500).json({ error:
            "Error al llamar al modelo"
        });
    }
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log('Servidor corriendo en el puerto ' + PORT);
});