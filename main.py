from simulate import simulate_system, plot_response

if __name__ == "__main__":
    t, y = simulate_system()
    plot_response(t, y)