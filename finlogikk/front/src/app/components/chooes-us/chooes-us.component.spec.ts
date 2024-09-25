import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChooesUsComponent } from './chooes-us.component';

describe('ChooesUsComponent', () => {
  let component: ChooesUsComponent;
  let fixture: ComponentFixture<ChooesUsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ChooesUsComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ChooesUsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
