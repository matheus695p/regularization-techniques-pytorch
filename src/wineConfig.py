import warnings
from argparse import ArgumentParser
warnings.filterwarnings("ignore", category=DeprecationWarning)


def arguments_parser():
    """
    El parser de argumentos de parámetros que hay que setiar para entrenar
    una red deep renewal
    Returns
    -------
    args : argparser
        argparser con todos los parámetros del modelo.
    """
    # argumentos
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--fff", help="haciendo weon a python", default="1")
    # agregar donde correr y guardar datos
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=80)
    parser.add_argument('--min_delta', type=float, default=1e-8)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--lr_factor', type=float, default=0.75)
    parser.add_argument('--lr_patience', type=int, default=15)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--validation_size', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=10e-3)
    parser.add_argument('--lr-scheduler', type=bool, default=True)
    parser.add_argument('--early-stopping', type=bool, default=True)
    parser.add_argument('--random-state', type=int, default=20)

    args = parser.parse_args()
    return args


def regularization_arguments():
    """
    El parser de argumentos de regularización
    Returns
    -------
    args : argparser
        argparser con parámetros de regularización.
    """
    parser = ArgumentParser()
    parser.add_argument(
        '--lr-scheduler', dest='lr_scheduler', action='store_true')
    parser.add_argument('--early-stopping',
                        dest='early_stopping', action='store_true')
    args = vars(parser.parse_args())
    return args
