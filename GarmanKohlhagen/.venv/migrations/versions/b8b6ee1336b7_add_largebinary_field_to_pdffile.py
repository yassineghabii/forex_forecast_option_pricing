from alembic import op
import sqlalchemy as sa

# Revision identifiers, used by Alembic.
revision = 'b8b6ee1336b7'
down_revision = 'e333408f4dba'
branch_labels = None
depends_on = None

def upgrade():
    # Upgrade logic only for the 'pdf_file' table
    with op.batch_alter_table('pdf_file', schema=None) as batch_op:
        batch_op.add_column(sa.Column('pdf_data', sa.LargeBinary(), nullable=True))

def downgrade():
    # Downgrade logic for the 'pdf_file' table
    with op.batch_alter_table('pdf_file', schema=None) as batch_op:
        batch_op.drop_column('pdf_data')
