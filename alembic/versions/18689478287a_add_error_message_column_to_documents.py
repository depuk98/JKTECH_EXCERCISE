"""add_error_message_column_to_documents

Revision ID: 18689478287a
Revises: 97c14d946ec6
Create Date: 2025-03-04 18:16:21.503336

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '18689478287a'
down_revision: Union[str, None] = '97c14d946ec6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add the error_message column to documents table
    op.add_column('documents', sa.Column('error_message', sa.String(), nullable=True))


def downgrade() -> None:
    # Remove the error_message column from documents table
    op.drop_column('documents', 'error_message')
