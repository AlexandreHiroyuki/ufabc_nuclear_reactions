import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Configuração para melhor visualização
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (15, 10)

# Constantes físicas
# Meias-vidas em anos
T_half_U234 = 245500
T_half_Th230 = 75380
T_half_Ra226 = 1600
T_half_Rn222 = 3.82/365

# Constantes de decaimento
lambda_U234 = np.log(2) / T_half_U234
lambda_Th230 = np.log(2) / T_half_Th230
lambda_Ra226 = np.log(2) / T_half_Ra226
lambda_Rn222 = np.log(2) / T_half_Rn222

# Condições iniciais
N0_U234 = 100000  # Número inicial de núcleos U-234
N0_Th230 = 0    # Assumindo que não há Th-230 inicial
N0_Ra226 = 0    # Assumindo que não há Ra-226 inicial
N0_Rn222 = 0    # Assumindo que não há Rn-222 inicial

# Tempo em anos
t = np.linspace(0, 500000, 10000)

def calculate_nuclei_population(t, lambda_parent, lambda_daughter, N0_parent, N0_daughter=0):
    """
    Calcula a população de núcleos usando a equação de Bateman para decaimento pai-filho.
    
    Args:
        t: array de tempo
        lambda_parent: constante de decaimento do pai
        lambda_daughter: constante de decaimento do filho
        N0_parent: população inicial do pai
        N0_daughter: população inicial do filho (padrão 0)
    
    Returns:
        N_parent, N_daughter: populações do pai e filho
    """
    # População do pai (decai exponencialmente)
    N_parent = N0_parent * np.exp(-lambda_parent * t)
    
    # População do filho (equação de Bateman)
    if lambda_parent != lambda_daughter:
        N_daughter = (lambda_parent * N0_parent / (lambda_daughter - lambda_parent)) * \
                     (np.exp(-lambda_parent * t) - np.exp(-lambda_daughter * t)) + \
                     N0_daughter * np.exp(-lambda_daughter * t)
    else:
        # Caso especial quando as constantes são iguais
        N_daughter = lambda_parent * N0_parent * t * np.exp(-lambda_parent * t) + \
                     N0_daughter * np.exp(-lambda_parent * t)
    
    return N_parent, N_daughter

def calculate_activity(N, lambda_decay):
    """
    Calcula a atividade radioativa A = λN.
    
    Args:
        N: população de núcleos
        lambda_decay: constante de decaimento
    
    Returns:
        A: atividade em decaimento/ano
    """
    return lambda_decay * N

# Cálculo das populações para cada par pai-filho
N_U234, N_Th230 = calculate_nuclei_population(t, lambda_U234, lambda_Th230, N0_U234, N0_Th230)
N_Th230_from_U, N_Ra226 = calculate_nuclei_population(t, lambda_Th230, lambda_Ra226, N_Th230, N0_Ra226)
N_Ra226_from_Th, N_Rn222 = calculate_nuclei_population(t, lambda_Ra226, lambda_Rn222, N_Ra226, N0_Rn222)

# Cálculo das atividades
A_U234 = calculate_activity(N_U234, lambda_U234)
A_Th230 = calculate_activity(N_Th230, lambda_Th230)
A_Ra226 = calculate_activity(N_Ra226_from_Th, lambda_Ra226)
A_Rn222 = calculate_activity(N_Rn222, lambda_Rn222)

# Criação dos gráficos
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])

# Gráfico 1: Populações N(t) - U-234 → Th-230
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t/1000, N_U234, 'b-', linewidth=2, label='U-234 (Pai)')
ax1.plot(t/1000, N_Th230, 'r-', linewidth=2, label='Th-230 (Filho)')
ax1.set_xlabel('Tempo (milhares de anos)')
ax1.set_ylabel('Número de Núcleos N(t)')
ax1.set_title('Decaimento U-234 → Th-230\nPopulação de Núcleos')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 500)

# Gráfico 2: Atividades A(t) - U-234 → Th-230
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(t/1000, A_U234, 'b--', linewidth=2, label='Atividade U-234')
ax2.plot(t/1000, A_Th230, 'r--', linewidth=2, label='Atividade Th-230')
ax2.set_xlabel('Tempo (milhares de anos)')
ax2.set_ylabel('Atividade A(t) (decai/ano)')
ax2.set_title('Decaimento U-234 → Th-230\nAtividade Radioativa')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 500)

# Gráfico 3: Populações N(t) - Ra-226 → Rn-222
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(t/1000, N_Ra226_from_Th, 'g-', linewidth=2, label='Ra-226 (Pai)')
ax3.plot(t/1000, N_Rn222, 'm-', linewidth=2, label='Rn-222 (Filho)')
ax3.set_xlabel('Tempo (milhares de anos)')
ax3.set_ylabel('Número de Núcleos N(t)')
ax3.set_title('Decaimento Ra-226 → Rn-222\nPopulação de Núcleos')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 10)

# Gráfico 4: Atividades A(t) - Ra-226 → Rn-222
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(t/1000, A_Ra226, 'g--', linewidth=3, label='Atividade Ra-226')
ax4.plot(t/1000, A_Rn222, 'm--', linewidth=1.5, label='Atividade Rn-222')
ax4.set_xlabel('Tempo (milhares de anos)')
ax4.set_ylabel('Atividade A(t) (decai/ano)')
ax4.set_title('Decaimento Ra-226 → Rn-222\nAtividade Radioativa')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 10)

plt.tight_layout()
plt.show()

print("=== ANÁLISE DO DECAIMENTO NUCLEAR ===")
print(f"Meia-vida U-234: {T_half_U234:,.0f} anos")
print(f"Meia-vida Th-230: {T_half_Th230:,.0f} anos")
print(f"Meia-vida Ra-226: {T_half_Ra226:,.0f} anos")
print(f"Meia-vida Rn-222: {T_half_Rn222:.4f} anos")
print(f"\nConstantes de decaimento:")
print(f"λ(U-234) = {lambda_U234:.2e} /ano")
print(f"λ(Th-230) = {lambda_Th230:.2e} /ano")
print(f"λ(Ra-226) = {lambda_Ra226:.2e} /ano")
print(f"λ(Rn-222) = {lambda_Rn222:.2e} /ano")

# Análise do equilíbrio secular
print(f"\n=== ANÁLISE DO EQUILÍBRIO ===")
print("No equilíbrio secular (t >> T_1/2 do pai):")
print(f"A(Th-230)/A(U-234) = {lambda_Th230/lambda_U234:.6f}")
print(f"A(Ra-226)/A(Th-230) = {lambda_Ra226/lambda_Th230:.6f}")
print(f"A(Rn-222)/A(Ra-226) = {lambda_Rn222/lambda_Ra226:.6f}")

# Tempo para atingir 99% do equilíbrio
t_equilibrium_Th = -np.log(0.01) / lambda_U234
t_equilibrium_Ra = -np.log(0.01) / lambda_Th230
t_equilibrium_Rn = -np.log(0.01) / lambda_Ra226

print(f"\nTempo para atingir 99% do equilíbrio:")
print(f"Th-230: {t_equilibrium_Th/1000:.1f} mil anos")
print(f"Ra-226: {t_equilibrium_Ra/1000:.1f} mil anos")
print(f"Rn-222: {t_equilibrium_Rn:.1f} anos")
