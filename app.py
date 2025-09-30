
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Simulador de Desajustes", layout="wide")
st.title("Simulador de Desajustes: Exceso de stock • Roturas • Última milla")

scenario = st.sidebar.selectbox("Escenario", ["Exceso de stock","Roturas de inventario","Retrasos en última milla"])

DAYS = st.sidebar.slider("Horizonte (días)", 14, 120, 45)
price = st.sidebar.number_input("Precio por unidad (€)", 1.0, 500.0, 20.0, step=1.0)
holding_cost = st.sidebar.number_input("Coste almacenamiento (€/unid/día)", 0.0, 5.0, 0.05, step=0.01)
penalty_stockout = st.sidebar.number_input("Penalización por rotura (€/unid)", 0.0, 100.0, 10.0, step=1.0)
transport_unit = st.sidebar.number_input("Coste transporte base (€/unid)", 0.1, 20.0, 1.0, step=0.1)
np.random.seed(17)

def run_simulation(scenario: str):
    base_demand = st.sidebar.number_input("Demanda media (unid/día)", 50, 20000, 1000, step=50)
    demand_cv = st.sidebar.slider("Variabilidad demanda (CV %)", 0, 120, 20, step=5)
    mu, sigma = base_demand, base_demand*(demand_cv/100)
    demand = np.maximum(0, np.random.normal(mu, sigma, DAYS).astype(int))

    inv = np.zeros(DAYS+1, dtype=int)
    backlog = np.zeros(DAYS+1, dtype=int)
    produced = np.zeros(DAYS, dtype=int)
    shipped = np.zeros(DAYS, dtype=int)
    delivered = np.zeros(DAYS, dtype=int)
    cost_storage = 0.0
    cost_transport = 0.0
    penalties = 0.0

    if scenario == "Exceso de stock":
        st.sidebar.header("Parámetros – Exceso de stock")
        prod_cap = st.sidebar.number_input("Capacidad de producción (unid/día)", 100, 50000, 1400, step=100)
        demand_shift = st.sidebar.slider("Caída de demanda por shock (%)", 0, 80, 30, step=5)
        overprod_bias = st.sidebar.slider("Sesgo de planificación (sobreproducción %)", 0, 100, 20, step=5)
        fleet_cap = st.sidebar.number_input("Capacidad de transporte (unid/día)", 100, 50000, 1200, step=100)

        demand = (demand * (1 - demand_shift/100)).astype(int)
        for t in range(DAYS):
            produced[t] = int(prod_cap * (1 + overprod_bias/100))
            inv[t] += produced[t]
            request = demand[t] + backlog[t]
            ship_today = min(inv[t], fleet_cap, request)
            shipped[t] = ship_today
            delivered[t] = ship_today
            inv[t] -= ship_today
            backlog[t+1] = max(0, request - ship_today)
            cost_storage += inv[t] * holding_cost
            cost_transport += shipped[t] * transport_unit

    elif scenario == "Roturas de inventario":
        st.sidebar.header("Parámetros – Roturas de inventario")
        prod_cap = st.sidebar.number_input("Capacidad de producción (unid/día)", 100, 50000, 900, step=100)
        supplier_delay_prob = st.sidebar.slider("Prob. retraso proveedor (%)", 0, 100, 30, step=5)
        delay_factor = st.sidebar.slider("Impacto del retraso en producción (%)", 0, 100, 40, step=5)
        fleet_cap = st.sidebar.number_input("Capacidad de transporte (unid/día)", 100, 50000, 1200, step=100)

        for t in range(DAYS):
            delay = np.random.rand() < (supplier_delay_prob/100)
            effective_cap = int(prod_cap * (1 - (delay_factor/100 if delay else 0)))
            produced[t] = effective_cap
            inv[t] += produced[t]
            request = demand[t] + backlog[t]
            ship_today = min(inv[t], fleet_cap, request)
            shipped[t] = ship_today
            delivered[t] = ship_today
            inv[t] -= ship_today
            backlog[t+1] = max(0, request - ship_today)
            cost_storage += inv[t] * holding_cost
            cost_transport += shipped[t] * transport_unit
            not_served = max(0, demand[t] - shipped[t])
            penalties += not_served * penalty_stockout

    else:  # Retrasos en última milla
        st.sidebar.header("Parámetros – Última milla")
        prod_cap = st.sidebar.number_input("Capacidad de producción (unid/día)", 100, 50000, 1200, step=100)
        fleet_cap = st.sidebar.number_input("Capacidad de transporte (unid/día)", 50, 50000, 700, step=50)
        traffic_prob = st.sidebar.slider("Prob. congestión (%)", 0, 100, 35, step=5)
        traffic_impact = st.sidebar.slider("Impacto congestión (↓capacidad %)", 0, 100, 40, step=5)
        extra_cost = st.sidebar.number_input("Sobrecoste por urgencia (€/unid)", 0.0, 50.0, 2.5, step=0.5)

        for t in range(DAYS):
            produced[t] = prod_cap
            inv[t] += produced[t]
            congested = np.random.rand() < (traffic_prob/100)
            effective_fleet = int(fleet_cap * (1 - (traffic_impact/100 if congested else 0)))
            ship_today = min(inv[t], effective_fleet, demand[t] + backlog[t])
            shipped[t] = ship_today
            delivered[t] = ship_today
            inv[t] -= ship_today
            backlog[t+1] = max(0, demand[t] + backlog[t] - ship_today)
            cost_storage += inv[t] * holding_cost
            base_cost = shipped[t] * transport_unit
            rush_units = max(0, (demand[t] - ship_today))
            cost_transport += base_cost + rush_units * extra_cost

    # KPIs
    service = (delivered.sum() / max(demand.sum(),1)) * 100
    inv_avg = np.mean(inv[:-1])
    backlog_end = backlog[-1]
    kpis = {
        "Nivel de servicio (%)": round(service,1),
        "Inventario medio (unid)": int(inv_avg),
        "Backlog final (unid)": int(backlog_end),
        "Coste almacenamiento (€)": round(cost_storage,2),
        "Coste transporte (€)": round(cost_transport,2),
        "Penalizaciones (€)": round(penalties,2),
        "Ingresos (€)": round(delivered.sum()*price,2)
    }

    df = pd.DataFrame({
        "día": np.arange(1, DAYS+1),
        "demanda": demand,
        "producido": produced,
        "enviado": shipped,
        "entregado": delivered,
        "inventario": inv[1:DAYS+1],
        "backlog": backlog[1:DAYS+1]
    })
    return kpis, df

kpis, df = run_simulation(scenario)

# KPIs
c1,c2,c3 = st.columns(3)
c1.metric("Nivel de servicio", f"{kpis['Nivel de servicio (%)']:.1f}%")
c2.metric("Inventario medio", f"{kpis['Inventario medio (unid)']}")
c3.metric("Backlog final", f"{kpis['Backlog final (unid)']}")

c4,c5,c6,c7 = st.columns(4)
c4.metric("Coste almacen.", f"{kpis['Coste almacenamiento (€)']} €")
c5.metric("Coste transporte", f"{kpis['Coste transporte (€)']} €")
c6.metric("Penalizaciones", f"{kpis['Penalizaciones (€)']} €")
c7.metric("Ingresos", f"{kpis['Ingresos (€)']} €")

# Charts
st.subheader("Series principales")
g1,g2 = st.columns(2)
with g1:
    st.plotly_chart(px.line(df, x="día", y=["demanda","producido"], title="Demanda vs Producción"), use_container_width=True)
with g2:
    st.plotly_chart(px.line(df, x="día", y=["enviado","entregado"], title="Enviado vs Entregado"), use_container_width=True)

h1,h2 = st.columns(2)
with h1:
    st.plotly_chart(px.line(df, x="día", y=["inventario","backlog"], title="Inventario y Backlog"), use_container_width=True)
with h2:
    st.plotly_chart(px.bar(df, x="día", y="entregado", title="Entregado por día"), use_container_width=True)

st.subheader("Tabla de resultados")
st.dataframe(df)
