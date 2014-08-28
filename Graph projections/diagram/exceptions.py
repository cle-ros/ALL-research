__author__ = 'clemens'


class DiagramException(Exception):
    """
    A general parent class for all diagram exceptions
    """
    pass


class NoSuchNodeException(DiagramException):
    """
    A slightly more fitting Exception for this usecase :-)
    """
    pass


class TerminalNodeException(DiagramException):
    """
    A slightly more fitting Exception for this usecase :-)
    """
    pass


class UnmatchingDiagramsException(DiagramException):
    """
    An exception for cases in which two diagrams do not match under some operation
    """
    pass