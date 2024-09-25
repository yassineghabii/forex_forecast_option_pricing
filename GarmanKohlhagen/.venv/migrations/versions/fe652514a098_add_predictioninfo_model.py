"""Add PredictionInfo model

Revision ID: fe652514a098
Revises: b8b6ee1336b7
Create Date: 2024-09-21 14:35:00.668312

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'fe652514a098'
down_revision = 'b8b6ee1336b7'
branch_labels = None
depends_on = None


def upgrade():
    # Creating the 'prediction_info' table
    op.create_table('prediction_info',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('ticker', sa.String(length=50), nullable=False),
        sa.Column('prediction_data', sa.JSON(), nullable=False),
        sa.Column('performance_metrics', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade():
    # Dropping the 'prediction_info' table in case of rollback
    op.drop_table('prediction_info')
