import sys
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('[ERROR]: usage is python convergence_plot.py <csv report>')
        sys.exit(1)

    data = pd.read_csv(sys.argv[1])

    plt.title('Convergence order (half both $\Delta x$ and $\Delta t$)')
    plt.xlabel('$\Delta x$')
    plt.ylabel('Error')

    plt.loglog(data['deltax'], data['l1'],        label='$L_1$')
    plt.loglog(data['deltax'], data['l2'],        label='$L_2$')
    plt.loglog(data['deltax'], data['linf'],      label='$L_{\infty}$')
    plt.loglog(data['deltax'], data['deltax']**2, label='$O(\Delta x^2)$')

    plt.legend()
    plt.savefig('convergence_order.png')
    plt.show()
