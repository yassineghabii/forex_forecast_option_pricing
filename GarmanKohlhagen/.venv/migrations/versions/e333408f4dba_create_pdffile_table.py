"""Create PDFFile table

Revision ID: e333408f4dba
Revises: 
Create Date: 2024-09-13 18:06:48.148369

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'e333408f4dba'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create the pdf_file table with necessary columns
    op.create_table(
        'pdf_file',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('filepath', sa.String(length=500), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True)
    )


def downgrade():
    # Drop the pdf_file table if the migration is rolled back
    op.drop_table('pdf_file')
