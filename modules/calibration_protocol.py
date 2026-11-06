"""Alvos físicos e pontos apresentados nas duas telas do experimento."""
import random

PROFILES = {
    "quick": {"duration": 3.0, "settling": 1.0},
    "standard": {"duration": 4.0, "settling": 0.5},
    "assisted": {"duration": 3.0, "settling": 1.0},
}


def calibration_plan(check=False, profile="standard"):
    if profile not in PROFILES:
        raise ValueError("Perfil de calibração inválido")
    def screen(target, x, y, description):
        return {"target": target, "point": description, "marker": [x, y],
                "speech": "Notebook. Siga o ponto." if target == "NOTEBOOK" else "TV. Siga o ponto."}
    notebook = [screen("NOTEBOOK", .3, .35, "o ponto à esquerda no notebook"),
                screen("NOTEBOOK", .7, .65, "o ponto à direita no notebook"),
                screen("NOTEBOOK", .5, .5, "o ponto central no notebook")]
    tv = [screen("TV", .3, .35, "o ponto à esquerda na TV"),
          screen("TV", .7, .65, "o ponto à direita na TV"),
          screen("TV", .5, .5, "o ponto central na TV")]
    if profile == "assisted":
        notebook = [dict(step, point=point, speech="Olhe para o notebook.") for step, point in zip(notebook, [
            "o centro da tela do notebook", "a parte superior da tela do notebook", "o centro da tela do notebook"])]
        tv = [dict(step, point=point, speech="Olhe para a TV.") for step, point in zip(tv, [
            "o centro da TV acima do notebook", "a parte inferior da TV", "o centro da TV acima do notebook"])]
    robot = [
        {"target": "NAO", "point": "a cabeça do NAO físico", "part": "head", "speech": "Cabeça do robô."},
        {"target": "NAO", "point": "a perna esquerda do NAO físico (à sua direita)",
         "part": "left-leg", "speech": "Perna do robô à sua direita."},
        {"target": "NAO", "point": "a perna direita do NAO físico (à sua esquerda)",
         "part": "right-leg", "speech": "Perna do robô à sua esquerda."},
    ]
    outside = [
        {"target": "OUTRO_LOCAL", "point": "um local à sua esquerda, fora das telas e do robô", "speech": "Olhe para fora, à esquerda."},
        {"target": "OUTRO_LOCAL", "point": "um local à sua direita, fora das telas e do robô", "speech": "Olhe para fora, à direita."},
        {"target": "OUTRO_LOCAL", "point": "a mesa, fora do notebook e do robô", "speech": "Olhe para a mesa, fora dos alvos."},
    ]
    if check:
        if profile in ("quick", "assisted"):
            # Verifica também as duas pernas; estes dados não ajustam o modelo.
            plan = [notebook[2], tv[2], *robot, outside[0]]
        else:
            plan = [notebook[0], notebook[1], tv[0], tv[1], *robot, outside[0]]
        random.SystemRandom().shuffle(plan)
        return plan
    if profile in ("quick", "assisted"):
        return [*notebook[:2], *tv[:2], *robot, *outside[:2]]
    return [*notebook, *tv, *robot, *outside]
