from .auth         import router as auth
from .accounts     import router as accounts
from .customer     import router as customer
from .employee     import router as employee
from .transactions import router as transactions

__all__ = [ "auth", "accounts", "customer", "employee", "transactions" ]
